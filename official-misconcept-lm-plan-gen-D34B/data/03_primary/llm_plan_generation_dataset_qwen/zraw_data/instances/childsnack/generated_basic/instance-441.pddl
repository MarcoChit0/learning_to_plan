; child-snack task with 3 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 334225

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child384 child371 child91 - child
    bread472 bread416 bread111 - bread-portion
    content150 content165 content65 - content-portion
    tray97 - tray
    table358 table211 table245 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray97 kitchen)
     (at_kitchen_bread bread472)
     (at_kitchen_bread bread416)
     (at_kitchen_bread bread111)
     (at_kitchen_content content150)
     (at_kitchen_content content165)
     (at_kitchen_content content65)
     (not_allergic_gluten child91)
     (not_allergic_gluten child384)
     (not_allergic_gluten child371)
     (waiting child384 table358)
     (waiting child371 table358)
     (waiting child91 table358)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child384)
     (served child371)
     (served child91)
    )
  )
)
