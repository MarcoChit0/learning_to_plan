; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 463329

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child159 child122 - child
    bread357 bread62 - bread-portion
    content205 content204 - content-portion
    tray271 tray87 - tray
    table312 table361 table349 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray271 kitchen)
     (at tray87 kitchen)
     (at_kitchen_bread bread357)
     (at_kitchen_bread bread62)
     (at_kitchen_content content205)
     (at_kitchen_content content204)
     (no_gluten_bread bread62)
     (no_gluten_content content205)
     (allergic_gluten child159)
     (not_allergic_gluten child122)
     (waiting child159 table361)
     (waiting child122 table312)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child159)
     (served child122)
    )
  )
)
