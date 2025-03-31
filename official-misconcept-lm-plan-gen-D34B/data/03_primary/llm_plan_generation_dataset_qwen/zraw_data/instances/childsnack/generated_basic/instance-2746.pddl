; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 31629

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child312 child165 - child
    bread202 bread412 - bread-portion
    content322 content274 - content-portion
    tray409 - tray
    table62 table331 table3 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray409 kitchen)
     (at_kitchen_bread bread202)
     (at_kitchen_bread bread412)
     (at_kitchen_content content322)
     (at_kitchen_content content274)
     (no_gluten_bread bread202)
     (no_gluten_content content322)
     (allergic_gluten child312)
     (not_allergic_gluten child165)
     (waiting child312 table331)
     (waiting child165 table3)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child312)
     (served child165)
    )
  )
)
