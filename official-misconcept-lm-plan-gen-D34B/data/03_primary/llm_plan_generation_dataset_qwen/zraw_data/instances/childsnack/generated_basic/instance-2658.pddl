; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 157910

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child168 - child
    bread273 - bread-portion
    content312 - content-portion
    tray216 tray114 tray307 tray165 - tray
    table323 table380 table306 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray216 kitchen)
     (at tray114 kitchen)
     (at tray307 kitchen)
     (at tray165 kitchen)
     (at_kitchen_bread bread273)
     (at_kitchen_content content312)
     (not_allergic_gluten child168)
     (waiting child168 table306)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child168)
    )
  )
)
