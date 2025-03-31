; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 331430

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child33 child23 - child
    bread166 bread387 - bread-portion
    content183 content234 - content-portion
    tray118 - tray
    table370 table390 table235 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray118 kitchen)
     (at_kitchen_bread bread166)
     (at_kitchen_bread bread387)
     (at_kitchen_content content183)
     (at_kitchen_content content234)
     (no_gluten_bread bread387)
     (no_gluten_content content234)
     (allergic_gluten child33)
     (not_allergic_gluten child23)
     (waiting child33 table390)
     (waiting child23 table370)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child33)
     (served child23)
    )
  )
)
