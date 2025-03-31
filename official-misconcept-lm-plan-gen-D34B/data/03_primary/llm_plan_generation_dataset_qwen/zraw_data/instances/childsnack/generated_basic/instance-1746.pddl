; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 975353

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child433 - child
    bread153 - bread-portion
    content360 - content-portion
    tray487 tray236 tray476 - tray
    table65 table32 table305 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray487 kitchen)
     (at tray236 kitchen)
     (at tray476 kitchen)
     (at_kitchen_bread bread153)
     (at_kitchen_content content360)
     (not_allergic_gluten child433)
     (waiting child433 table305)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child433)
    )
  )
)
