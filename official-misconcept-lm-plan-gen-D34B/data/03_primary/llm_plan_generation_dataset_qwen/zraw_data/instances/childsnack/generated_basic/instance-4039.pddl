; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 483816

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child79 child363 child155 - child
    bread54 bread20 bread58 - bread-portion
    content242 content282 content41 - content-portion
    tray119 - tray
    table366 table242 table280 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray119 kitchen)
     (at_kitchen_bread bread54)
     (at_kitchen_bread bread20)
     (at_kitchen_bread bread58)
     (at_kitchen_content content242)
     (at_kitchen_content content282)
     (at_kitchen_content content41)
     (not_allergic_gluten child363)
     (not_allergic_gluten child79)
     (not_allergic_gluten child155)
     (waiting child79 table280)
     (waiting child363 table280)
     (waiting child155 table280)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child79)
     (served child363)
     (served child155)
    )
  )
)
