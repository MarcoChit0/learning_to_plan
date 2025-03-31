; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 753968

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child428 - child
    bread96 - bread-portion
    content374 - content-portion
    tray182 tray308 tray117 - tray
    table397 table460 table100 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray182 kitchen)
     (at tray308 kitchen)
     (at tray117 kitchen)
     (at_kitchen_bread bread96)
     (at_kitchen_content content374)
     (not_allergic_gluten child428)
     (waiting child428 table100)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child428)
    )
  )
)
