; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 702234

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child164 child109 - child
    bread30 bread459 - bread-portion
    content203 content440 - content-portion
    tray363 tray263 - tray
    table117 table214 table473 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray363 kitchen)
     (at tray263 kitchen)
     (at_kitchen_bread bread30)
     (at_kitchen_bread bread459)
     (at_kitchen_content content203)
     (at_kitchen_content content440)
     (no_gluten_bread bread459)
     (no_gluten_content content203)
     (allergic_gluten child109)
     (not_allergic_gluten child164)
     (waiting child164 table117)
     (waiting child109 table117)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child164)
     (served child109)
    )
  )
)
