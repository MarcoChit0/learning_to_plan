; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 896702

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child409 child291 - child
    bread116 bread365 - bread-portion
    content198 content273 - content-portion
    tray207 - tray
    table391 table362 table187 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray207 kitchen)
     (at_kitchen_bread bread116)
     (at_kitchen_bread bread365)
     (at_kitchen_content content198)
     (at_kitchen_content content273)
     (no_gluten_bread bread365)
     (no_gluten_content content273)
     (allergic_gluten child291)
     (not_allergic_gluten child409)
     (waiting child409 table391)
     (waiting child291 table391)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child409)
     (served child291)
    )
  )
)
