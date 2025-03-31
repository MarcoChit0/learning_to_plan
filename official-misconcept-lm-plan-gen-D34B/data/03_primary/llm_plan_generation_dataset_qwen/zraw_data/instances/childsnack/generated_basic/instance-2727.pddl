; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 289173

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child261 child331 child435 - child
    bread147 bread146 bread401 - bread-portion
    content305 content173 content23 - content-portion
    tray350 tray279 tray222 - tray
    table175 table242 table6 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray350 kitchen)
     (at tray279 kitchen)
     (at tray222 kitchen)
     (at_kitchen_bread bread147)
     (at_kitchen_bread bread146)
     (at_kitchen_bread bread401)
     (at_kitchen_content content305)
     (at_kitchen_content content173)
     (at_kitchen_content content23)
     (not_allergic_gluten child261)
     (not_allergic_gluten child331)
     (not_allergic_gluten child435)
     (waiting child261 table242)
     (waiting child331 table175)
     (waiting child435 table175)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child261)
     (served child331)
     (served child435)
    )
  )
)
