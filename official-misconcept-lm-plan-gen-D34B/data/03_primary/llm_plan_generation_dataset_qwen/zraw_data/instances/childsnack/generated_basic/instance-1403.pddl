; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 579854

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child435 child155 - child
    bread201 bread36 - bread-portion
    content360 content113 - content-portion
    tray140 - tray
    table181 table173 table207 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray140 kitchen)
     (at_kitchen_bread bread201)
     (at_kitchen_bread bread36)
     (at_kitchen_content content360)
     (at_kitchen_content content113)
     (no_gluten_bread bread36)
     (no_gluten_content content360)
     (allergic_gluten child155)
     (not_allergic_gluten child435)
     (waiting child435 table207)
     (waiting child155 table173)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child435)
     (served child155)
    )
  )
)
