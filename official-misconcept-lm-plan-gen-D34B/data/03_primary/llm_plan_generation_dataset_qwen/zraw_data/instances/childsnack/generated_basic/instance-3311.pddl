; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.3
; random seed: 579283

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child266 child257 - child
    bread387 bread311 - bread-portion
    content207 content401 - content-portion
    tray194 tray249 - tray
    table242 table361 table293 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray194 kitchen)
     (at tray249 kitchen)
     (at_kitchen_bread bread387)
     (at_kitchen_bread bread311)
     (at_kitchen_content content207)
     (at_kitchen_content content401)
     (not_allergic_gluten child257)
     (not_allergic_gluten child266)
     (waiting child266 table361)
     (waiting child257 table242)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child266)
     (served child257)
    )
  )
)
