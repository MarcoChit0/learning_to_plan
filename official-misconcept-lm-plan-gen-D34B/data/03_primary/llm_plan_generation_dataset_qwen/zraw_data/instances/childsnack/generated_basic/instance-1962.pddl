; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 760023

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child278 child291 - child
    bread239 bread103 - bread-portion
    content319 content136 - content-portion
    tray467 - tray
    table278 table289 table276 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray467 kitchen)
     (at_kitchen_bread bread239)
     (at_kitchen_bread bread103)
     (at_kitchen_content content319)
     (at_kitchen_content content136)
     (no_gluten_bread bread239)
     (no_gluten_content content319)
     (allergic_gluten child278)
     (not_allergic_gluten child291)
     (waiting child278 table289)
     (waiting child291 table278)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child278)
     (served child291)
    )
  )
)
