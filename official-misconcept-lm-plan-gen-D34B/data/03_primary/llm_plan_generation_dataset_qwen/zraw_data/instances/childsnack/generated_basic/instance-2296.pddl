; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 443429

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child116 child111 - child
    bread29 bread420 - bread-portion
    content316 content68 - content-portion
    tray467 - tray
    table108 table259 table33 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray467 kitchen)
     (at_kitchen_bread bread29)
     (at_kitchen_bread bread420)
     (at_kitchen_content content316)
     (at_kitchen_content content68)
     (no_gluten_bread bread29)
     (no_gluten_content content68)
     (allergic_gluten child111)
     (not_allergic_gluten child116)
     (waiting child116 table108)
     (waiting child111 table108)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child116)
     (served child111)
    )
  )
)
