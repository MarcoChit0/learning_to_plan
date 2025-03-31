; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 597336

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child151 child129 - child
    bread176 bread14 - bread-portion
    content387 content100 - content-portion
    tray155 tray303 - tray
    table325 table449 table139 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray155 kitchen)
     (at tray303 kitchen)
     (at_kitchen_bread bread176)
     (at_kitchen_bread bread14)
     (at_kitchen_content content387)
     (at_kitchen_content content100)
     (not_allergic_gluten child151)
     (not_allergic_gluten child129)
     (waiting child151 table139)
     (waiting child129 table325)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child151)
     (served child129)
    )
  )
)
