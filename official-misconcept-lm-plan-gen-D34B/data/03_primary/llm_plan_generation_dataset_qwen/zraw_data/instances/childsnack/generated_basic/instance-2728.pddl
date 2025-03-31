; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 880444

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child185 child111 child316 - child
    bread204 bread466 bread10 - bread-portion
    content100 content287 content202 - content-portion
    tray217 tray207 tray342 - tray
    table211 table269 table181 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray217 kitchen)
     (at tray207 kitchen)
     (at tray342 kitchen)
     (at_kitchen_bread bread204)
     (at_kitchen_bread bread466)
     (at_kitchen_bread bread10)
     (at_kitchen_content content100)
     (at_kitchen_content content287)
     (at_kitchen_content content202)
     (not_allergic_gluten child111)
     (not_allergic_gluten child185)
     (not_allergic_gluten child316)
     (waiting child185 table211)
     (waiting child111 table181)
     (waiting child316 table269)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child185)
     (served child111)
     (served child316)
    )
  )
)
