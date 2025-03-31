; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 520583

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child1 child386 - child
    bread3 bread53 - bread-portion
    content86 content225 - content-portion
    tray335 tray145 tray446 - tray
    table321 table245 table359 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray335 kitchen)
     (at tray145 kitchen)
     (at tray446 kitchen)
     (at_kitchen_bread bread3)
     (at_kitchen_bread bread53)
     (at_kitchen_content content86)
     (at_kitchen_content content225)
     (no_gluten_bread bread53)
     (no_gluten_content content86)
     (allergic_gluten child386)
     (not_allergic_gluten child1)
     (waiting child1 table245)
     (waiting child386 table245)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child1)
     (served child386)
    )
  )
)
