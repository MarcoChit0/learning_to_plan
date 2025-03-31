; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 670488

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child399 - child
    bread160 - bread-portion
    content367 - content-portion
    tray338 - tray
    table220 table27 table266 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray338 kitchen)
     (at_kitchen_bread bread160)
     (at_kitchen_content content367)
     (not_allergic_gluten child399)
     (waiting child399 table266)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child399)
    )
  )
)
