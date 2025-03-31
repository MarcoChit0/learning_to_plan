; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 914257

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child245 child456 - child
    bread11 bread471 - bread-portion
    content453 content33 - content-portion
    tray372 tray264 - tray
    table143 table452 table93 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray372 kitchen)
     (at tray264 kitchen)
     (at_kitchen_bread bread11)
     (at_kitchen_bread bread471)
     (at_kitchen_content content453)
     (at_kitchen_content content33)
     (no_gluten_bread bread11)
     (no_gluten_content content33)
     (allergic_gluten child456)
     (not_allergic_gluten child245)
     (waiting child245 table143)
     (waiting child456 table93)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child245)
     (served child456)
    )
  )
)
