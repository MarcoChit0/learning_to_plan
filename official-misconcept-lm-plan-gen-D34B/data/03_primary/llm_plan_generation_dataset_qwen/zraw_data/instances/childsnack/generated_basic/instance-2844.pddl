; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 713164

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child219 - child
    bread199 - bread-portion
    content334 - content-portion
    tray353 tray447 - tray
    table246 table142 table285 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray353 kitchen)
     (at tray447 kitchen)
     (at_kitchen_bread bread199)
     (at_kitchen_content content334)
     (not_allergic_gluten child219)
     (waiting child219 table285)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child219)
    )
  )
)
