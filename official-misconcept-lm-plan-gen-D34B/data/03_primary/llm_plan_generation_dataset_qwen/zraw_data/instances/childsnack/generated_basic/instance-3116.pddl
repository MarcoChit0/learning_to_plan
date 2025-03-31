; child-snack task with 2 children and 0.51 gluten factor 
; constant factor of 1.9
; random seed: 403381

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child176 child434 - child
    bread228 bread283 - bread-portion
    content332 content175 - content-portion
    tray470 tray361 tray410 - tray
    table290 table223 table391 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray470 kitchen)
     (at tray361 kitchen)
     (at tray410 kitchen)
     (at_kitchen_bread bread228)
     (at_kitchen_bread bread283)
     (at_kitchen_content content332)
     (at_kitchen_content content175)
     (no_gluten_bread bread228)
     (no_gluten_content content332)
     (allergic_gluten child434)
     (not_allergic_gluten child176)
     (waiting child176 table290)
     (waiting child434 table391)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child176)
     (served child434)
    )
  )
)
