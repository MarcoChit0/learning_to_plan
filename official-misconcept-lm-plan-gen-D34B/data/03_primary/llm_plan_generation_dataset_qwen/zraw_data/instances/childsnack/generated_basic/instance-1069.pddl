; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 681106

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child105 child190 - child
    bread197 bread53 - bread-portion
    content259 content63 - content-portion
    tray388 tray362 tray474 - tray
    table102 table126 table351 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray388 kitchen)
     (at tray362 kitchen)
     (at tray474 kitchen)
     (at_kitchen_bread bread197)
     (at_kitchen_bread bread53)
     (at_kitchen_content content259)
     (at_kitchen_content content63)
     (not_allergic_gluten child105)
     (not_allergic_gluten child190)
     (waiting child105 table126)
     (waiting child190 table126)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child105)
     (served child190)
    )
  )
)
