; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 603358

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child336 child424 - child
    bread458 bread290 - bread-portion
    content322 content52 - content-portion
    tray367 tray383 tray450 - tray
    table364 table123 table476 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray367 kitchen)
     (at tray383 kitchen)
     (at tray450 kitchen)
     (at_kitchen_bread bread458)
     (at_kitchen_bread bread290)
     (at_kitchen_content content322)
     (at_kitchen_content content52)
     (no_gluten_bread bread458)
     (no_gluten_content content322)
     (allergic_gluten child424)
     (not_allergic_gluten child336)
     (waiting child336 table123)
     (waiting child424 table476)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child336)
     (served child424)
    )
  )
)
