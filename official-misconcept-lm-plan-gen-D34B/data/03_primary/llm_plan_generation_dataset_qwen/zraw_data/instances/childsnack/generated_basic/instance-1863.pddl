; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 503315

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child159 child201 - child
    bread226 bread302 - bread-portion
    content223 content167 - content-portion
    tray30 tray87 - tray
    table496 table196 table179 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray30 kitchen)
     (at tray87 kitchen)
     (at_kitchen_bread bread226)
     (at_kitchen_bread bread302)
     (at_kitchen_content content223)
     (at_kitchen_content content167)
     (no_gluten_bread bread226)
     (no_gluten_content content223)
     (allergic_gluten child201)
     (not_allergic_gluten child159)
     (waiting child159 table496)
     (waiting child201 table496)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child159)
     (served child201)
    )
  )
)
