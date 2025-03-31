; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 923801

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child371 child106 - child
    bread322 bread245 - bread-portion
    content53 content188 - content-portion
    tray115 - tray
    table492 table228 table187 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray115 kitchen)
     (at_kitchen_bread bread322)
     (at_kitchen_bread bread245)
     (at_kitchen_content content53)
     (at_kitchen_content content188)
     (no_gluten_bread bread245)
     (no_gluten_content content53)
     (allergic_gluten child371)
     (not_allergic_gluten child106)
     (waiting child371 table187)
     (waiting child106 table187)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child371)
     (served child106)
    )
  )
)
