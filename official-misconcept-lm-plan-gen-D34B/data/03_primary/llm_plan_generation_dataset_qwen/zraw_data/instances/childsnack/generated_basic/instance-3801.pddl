; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 586658

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child99 - child
    bread123 - bread-portion
    content62 - content-portion
    tray267 - tray
    table96 table274 table200 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray267 kitchen)
     (at_kitchen_bread bread123)
     (at_kitchen_content content62)
     (not_allergic_gluten child99)
     (waiting child99 table274)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child99)
    )
  )
)
