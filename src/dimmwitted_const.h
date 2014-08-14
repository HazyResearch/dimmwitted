
#ifndef _DIMMWITTED_CONST_H
#define _DIMMWITTED_CONST_H

/**
 * Model replication strategy
 * to use in DimmWitted.
 */
enum ModelReplType{
  DW_MODELREPL_PERCORE,
  DW_MODELREPL_PERNODE,
  DW_MODELREPL_PERMACHINE,
  DW_MODELREPL_SINGLETHREAD_DEBUG
};

/**
 * Data replication strategy
 * to use in DimmWitted.
 */
enum DataReplType{
  DW_DATAREPL_FULL,
  DW_DATAREPL_SHARDING
};

/**
 * Access method to use
 * in DimmWitted.
 */
enum AccessMode{
  DW_ACCESS_ROW,
  DW_ACCESS_COL,
  DW_ACCESS_C2R
};




#endif