; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 964552

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child164 child319 - child
    bread278 bread192 - bread-portion
    content86 content242 - content-portion
    tray380 tray229 - tray
    table387 table155 table11 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray380 kitchen)
     (at tray229 kitchen)
     (at_kitchen_bread bread278)
     (at_kitchen_bread bread192)
     (at_kitchen_content content86)
     (at_kitchen_content content242)
     (no_gluten_bread bread192)
     (no_gluten_content content242)
     (allergic_gluten child319)
     (not_allergic_gluten child164)
     (waiting child164 table155)
     (waiting child319 table387)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child164)
     (served child319)
    )
  )
)
