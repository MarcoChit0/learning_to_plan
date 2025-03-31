; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 589286

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child305 child83 child66 - child
    bread32 bread298 bread163 - bread-portion
    content338 content81 content42 - content-portion
    tray256 - tray
    table45 table132 table302 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray256 kitchen)
     (at_kitchen_bread bread32)
     (at_kitchen_bread bread298)
     (at_kitchen_bread bread163)
     (at_kitchen_content content338)
     (at_kitchen_content content81)
     (at_kitchen_content content42)
     (not_allergic_gluten child83)
     (not_allergic_gluten child305)
     (not_allergic_gluten child66)
     (waiting child305 table45)
     (waiting child83 table45)
     (waiting child66 table302)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child305)
     (served child83)
     (served child66)
    )
  )
)
