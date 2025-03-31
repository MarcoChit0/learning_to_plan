; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 321943

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child105 - child
    bread258 - bread-portion
    content5 - content-portion
    tray45 tray408 tray173 tray91 - tray
    table312 table438 table288 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray45 kitchen)
     (at tray408 kitchen)
     (at tray173 kitchen)
     (at tray91 kitchen)
     (at_kitchen_bread bread258)
     (at_kitchen_content content5)
     (not_allergic_gluten child105)
     (waiting child105 table438)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child105)
    )
  )
)
