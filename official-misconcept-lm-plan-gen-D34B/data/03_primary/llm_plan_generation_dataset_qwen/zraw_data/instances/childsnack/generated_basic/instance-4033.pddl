; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 164497

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child64 child263 child438 - child
    bread421 bread363 bread286 - bread-portion
    content401 content277 content180 - content-portion
    tray86 - tray
    table192 table422 table336 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray86 kitchen)
     (at_kitchen_bread bread421)
     (at_kitchen_bread bread363)
     (at_kitchen_bread bread286)
     (at_kitchen_content content401)
     (at_kitchen_content content277)
     (at_kitchen_content content180)
     (not_allergic_gluten child64)
     (not_allergic_gluten child263)
     (not_allergic_gluten child438)
     (waiting child64 table192)
     (waiting child263 table336)
     (waiting child438 table336)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child64)
     (served child263)
     (served child438)
    )
  )
)
