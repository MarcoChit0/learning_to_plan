; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 545177

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child290 child126 - child
    bread153 bread221 - bread-portion
    content64 content336 - content-portion
    tray103 tray210 - tray
    table204 table448 table320 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray103 kitchen)
     (at tray210 kitchen)
     (at_kitchen_bread bread153)
     (at_kitchen_bread bread221)
     (at_kitchen_content content64)
     (at_kitchen_content content336)
     (no_gluten_bread bread221)
     (no_gluten_content content336)
     (allergic_gluten child290)
     (not_allergic_gluten child126)
     (waiting child290 table448)
     (waiting child126 table204)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child290)
     (served child126)
    )
  )
)
