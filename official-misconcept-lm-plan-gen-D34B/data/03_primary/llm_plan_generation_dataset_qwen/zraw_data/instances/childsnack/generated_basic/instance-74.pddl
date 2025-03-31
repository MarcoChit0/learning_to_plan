; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 208968

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child120 child244 - child
    bread459 bread383 - bread-portion
    content312 content66 - content-portion
    tray154 tray67 - tray
    table83 table406 table436 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray154 kitchen)
     (at tray67 kitchen)
     (at_kitchen_bread bread459)
     (at_kitchen_bread bread383)
     (at_kitchen_content content312)
     (at_kitchen_content content66)
     (not_allergic_gluten child244)
     (not_allergic_gluten child120)
     (waiting child120 table406)
     (waiting child244 table436)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child120)
     (served child244)
    )
  )
)
