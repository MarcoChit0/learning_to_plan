; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 994129

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child398 child456 - child
    bread23 bread220 - bread-portion
    content289 content400 - content-portion
    tray220 tray310 tray438 - tray
    table437 table252 table282 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray220 kitchen)
     (at tray310 kitchen)
     (at tray438 kitchen)
     (at_kitchen_bread bread23)
     (at_kitchen_bread bread220)
     (at_kitchen_content content289)
     (at_kitchen_content content400)
     (no_gluten_bread bread220)
     (no_gluten_content content289)
     (allergic_gluten child398)
     (not_allergic_gluten child456)
     (waiting child398 table252)
     (waiting child456 table437)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child398)
     (served child456)
    )
  )
)
