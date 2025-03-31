; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 966614

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child391 child344 child266 - child
    bread451 bread96 bread227 - bread-portion
    content275 content395 content235 - content-portion
    tray309 tray400 tray277 - tray
    table380 table399 table413 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray309 kitchen)
     (at tray400 kitchen)
     (at tray277 kitchen)
     (at_kitchen_bread bread451)
     (at_kitchen_bread bread96)
     (at_kitchen_bread bread227)
     (at_kitchen_content content275)
     (at_kitchen_content content395)
     (at_kitchen_content content235)
     (not_allergic_gluten child391)
     (not_allergic_gluten child266)
     (not_allergic_gluten child344)
     (waiting child391 table380)
     (waiting child344 table413)
     (waiting child266 table380)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child391)
     (served child344)
     (served child266)
    )
  )
)
