; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 659084

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child132 child444 - child
    bread349 bread379 - bread-portion
    content289 content301 - content-portion
    tray338 - tray
    table74 table6 table380 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray338 kitchen)
     (at_kitchen_bread bread349)
     (at_kitchen_bread bread379)
     (at_kitchen_content content289)
     (at_kitchen_content content301)
     (no_gluten_bread bread379)
     (no_gluten_content content289)
     (allergic_gluten child132)
     (not_allergic_gluten child444)
     (waiting child132 table6)
     (waiting child444 table380)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child132)
     (served child444)
    )
  )
)
