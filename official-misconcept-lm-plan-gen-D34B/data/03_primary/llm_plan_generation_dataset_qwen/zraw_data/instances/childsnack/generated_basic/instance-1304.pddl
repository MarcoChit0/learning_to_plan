; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.5
; random seed: 601745

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child407 child353 - child
    bread459 bread82 - bread-portion
    content16 content14 - content-portion
    tray368 tray477 tray311 tray96 - tray
    table220 table491 table109 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray368 kitchen)
     (at tray477 kitchen)
     (at tray311 kitchen)
     (at tray96 kitchen)
     (at_kitchen_bread bread459)
     (at_kitchen_bread bread82)
     (at_kitchen_content content16)
     (at_kitchen_content content14)
     (no_gluten_bread bread459)
     (no_gluten_content content16)
     (allergic_gluten child353)
     (not_allergic_gluten child407)
     (waiting child407 table491)
     (waiting child353 table109)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child407)
     (served child353)
    )
  )
)
