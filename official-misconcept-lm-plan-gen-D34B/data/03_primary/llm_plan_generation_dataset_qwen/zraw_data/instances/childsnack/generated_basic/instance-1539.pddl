; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 123801

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child21 child47 - child
    bread97 bread92 - bread-portion
    content360 content87 - content-portion
    tray298 tray118 tray263 tray1 - tray
    table449 table125 table34 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray298 kitchen)
     (at tray118 kitchen)
     (at tray263 kitchen)
     (at tray1 kitchen)
     (at_kitchen_bread bread97)
     (at_kitchen_bread bread92)
     (at_kitchen_content content360)
     (at_kitchen_content content87)
     (not_allergic_gluten child47)
     (not_allergic_gluten child21)
     (waiting child21 table34)
     (waiting child47 table449)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child21)
     (served child47)
    )
  )
)
