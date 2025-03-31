; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 933918

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child93 child239 - child
    bread288 bread30 - bread-portion
    content407 content81 - content-portion
    tray34 tray207 tray130 - tray
    table428 table151 table437 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray34 kitchen)
     (at tray207 kitchen)
     (at tray130 kitchen)
     (at_kitchen_bread bread288)
     (at_kitchen_bread bread30)
     (at_kitchen_content content407)
     (at_kitchen_content content81)
     (no_gluten_bread bread30)
     (no_gluten_content content407)
     (allergic_gluten child93)
     (not_allergic_gluten child239)
     (waiting child93 table437)
     (waiting child239 table428)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child93)
     (served child239)
    )
  )
)
