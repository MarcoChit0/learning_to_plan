; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 772792

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child88 child447 - child
    bread368 bread350 - bread-portion
    content466 content401 - content-portion
    tray195 tray260 - tray
    table204 table172 table472 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray195 kitchen)
     (at tray260 kitchen)
     (at_kitchen_bread bread368)
     (at_kitchen_bread bread350)
     (at_kitchen_content content466)
     (at_kitchen_content content401)
     (no_gluten_bread bread350)
     (no_gluten_content content401)
     (allergic_gluten child88)
     (not_allergic_gluten child447)
     (waiting child88 table472)
     (waiting child447 table472)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child88)
     (served child447)
    )
  )
)
