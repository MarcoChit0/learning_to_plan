; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 593392

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child101 child141 child400 - child
    bread372 bread66 bread18 - bread-portion
    content344 content416 content103 - content-portion
    tray121 - tray
    table464 table17 table416 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray121 kitchen)
     (at_kitchen_bread bread372)
     (at_kitchen_bread bread66)
     (at_kitchen_bread bread18)
     (at_kitchen_content content344)
     (at_kitchen_content content416)
     (at_kitchen_content content103)
     (not_allergic_gluten child400)
     (not_allergic_gluten child101)
     (not_allergic_gluten child141)
     (waiting child101 table464)
     (waiting child141 table416)
     (waiting child400 table17)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child101)
     (served child141)
     (served child400)
    )
  )
)
