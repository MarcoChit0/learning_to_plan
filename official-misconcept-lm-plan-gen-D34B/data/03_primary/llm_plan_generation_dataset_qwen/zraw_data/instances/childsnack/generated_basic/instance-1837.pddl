; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 862081

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child463 child267 - child
    bread186 bread175 - bread-portion
    content37 content52 - content-portion
    tray337 - tray
    table170 table67 table334 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray337 kitchen)
     (at_kitchen_bread bread186)
     (at_kitchen_bread bread175)
     (at_kitchen_content content37)
     (at_kitchen_content content52)
     (no_gluten_bread bread175)
     (no_gluten_content content37)
     (allergic_gluten child267)
     (not_allergic_gluten child463)
     (waiting child463 table67)
     (waiting child267 table67)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child463)
     (served child267)
    )
  )
)
