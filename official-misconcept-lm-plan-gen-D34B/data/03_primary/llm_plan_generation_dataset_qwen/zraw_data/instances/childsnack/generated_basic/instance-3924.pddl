; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 482315

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child41 child146 - child
    bread437 bread191 - bread-portion
    content204 content356 - content-portion
    tray229 - tray
    table69 table73 table270 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray229 kitchen)
     (at_kitchen_bread bread437)
     (at_kitchen_bread bread191)
     (at_kitchen_content content204)
     (at_kitchen_content content356)
     (no_gluten_bread bread437)
     (no_gluten_content content356)
     (allergic_gluten child41)
     (not_allergic_gluten child146)
     (waiting child41 table270)
     (waiting child146 table69)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child41)
     (served child146)
    )
  )
)
