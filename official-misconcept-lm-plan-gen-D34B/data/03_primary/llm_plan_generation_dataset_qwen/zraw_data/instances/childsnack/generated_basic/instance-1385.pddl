; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 665478

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child364 child27 - child
    bread116 bread311 - bread-portion
    content191 content265 - content-portion
    tray477 - tray
    table492 table332 table474 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray477 kitchen)
     (at_kitchen_bread bread116)
     (at_kitchen_bread bread311)
     (at_kitchen_content content191)
     (at_kitchen_content content265)
     (no_gluten_bread bread116)
     (no_gluten_content content265)
     (allergic_gluten child27)
     (not_allergic_gluten child364)
     (waiting child364 table492)
     (waiting child27 table492)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child364)
     (served child27)
    )
  )
)
