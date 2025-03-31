; child-snack task with 2 children and 0.62 gluten factor 
; constant factor of 1.3
; random seed: 458449

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child329 child435 - child
    bread27 bread59 - bread-portion
    content40 content422 - content-portion
    tray332 tray335 tray389 tray403 - tray
    table67 table462 table83 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray332 kitchen)
     (at tray335 kitchen)
     (at tray389 kitchen)
     (at tray403 kitchen)
     (at_kitchen_bread bread27)
     (at_kitchen_bread bread59)
     (at_kitchen_content content40)
     (at_kitchen_content content422)
     (no_gluten_bread bread59)
     (no_gluten_content content40)
     (allergic_gluten child435)
     (not_allergic_gluten child329)
     (waiting child329 table83)
     (waiting child435 table67)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child329)
     (served child435)
    )
  )
)
