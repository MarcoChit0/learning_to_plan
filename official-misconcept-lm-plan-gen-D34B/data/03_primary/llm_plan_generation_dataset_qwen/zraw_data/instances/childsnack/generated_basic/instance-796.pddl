; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 972495

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child63 - child
    bread324 - bread-portion
    content60 - content-portion
    tray84 tray476 - tray
    table142 table320 table53 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray84 kitchen)
     (at tray476 kitchen)
     (at_kitchen_bread bread324)
     (at_kitchen_content content60)
     (not_allergic_gluten child63)
     (waiting child63 table320)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child63)
    )
  )
)
