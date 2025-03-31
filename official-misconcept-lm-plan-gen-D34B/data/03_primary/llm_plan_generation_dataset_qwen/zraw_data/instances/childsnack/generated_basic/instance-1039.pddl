; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 840193

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child211 child498 - child
    bread360 bread38 - bread-portion
    content357 content212 - content-portion
    tray443 tray178 tray412 - tray
    table250 table38 table423 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray443 kitchen)
     (at tray178 kitchen)
     (at tray412 kitchen)
     (at_kitchen_bread bread360)
     (at_kitchen_bread bread38)
     (at_kitchen_content content357)
     (at_kitchen_content content212)
     (not_allergic_gluten child211)
     (not_allergic_gluten child498)
     (waiting child211 table38)
     (waiting child498 table250)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child211)
     (served child498)
    )
  )
)
