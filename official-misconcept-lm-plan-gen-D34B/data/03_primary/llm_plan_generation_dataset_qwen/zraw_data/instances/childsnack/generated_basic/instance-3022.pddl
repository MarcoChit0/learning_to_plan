; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 9628

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child378 child454 - child
    bread306 bread121 - bread-portion
    content172 content480 - content-portion
    tray76 tray456 tray277 tray45 - tray
    table177 table18 table271 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray76 kitchen)
     (at tray456 kitchen)
     (at tray277 kitchen)
     (at tray45 kitchen)
     (at_kitchen_bread bread306)
     (at_kitchen_bread bread121)
     (at_kitchen_content content172)
     (at_kitchen_content content480)
     (not_allergic_gluten child454)
     (not_allergic_gluten child378)
     (waiting child378 table18)
     (waiting child454 table18)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child378)
     (served child454)
    )
  )
)
