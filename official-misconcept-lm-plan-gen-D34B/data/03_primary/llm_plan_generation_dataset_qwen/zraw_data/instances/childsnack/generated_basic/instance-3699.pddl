; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 965386

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child94 child290 - child
    bread63 bread185 - bread-portion
    content336 content160 - content-portion
    tray404 tray381 - tray
    table139 table369 table193 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray404 kitchen)
     (at tray381 kitchen)
     (at_kitchen_bread bread63)
     (at_kitchen_bread bread185)
     (at_kitchen_content content336)
     (at_kitchen_content content160)
     (no_gluten_bread bread63)
     (no_gluten_content content336)
     (allergic_gluten child290)
     (not_allergic_gluten child94)
     (waiting child94 table139)
     (waiting child290 table139)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child94)
     (served child290)
    )
  )
)
