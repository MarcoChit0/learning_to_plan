; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 960441

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child345 child153 child165 - child
    bread25 bread12 bread198 - bread-portion
    content64 content184 content392 - content-portion
    tray61 tray343 tray307 - tray
    table131 table91 table31 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray61 kitchen)
     (at tray343 kitchen)
     (at tray307 kitchen)
     (at_kitchen_bread bread25)
     (at_kitchen_bread bread12)
     (at_kitchen_bread bread198)
     (at_kitchen_content content64)
     (at_kitchen_content content184)
     (at_kitchen_content content392)
     (not_allergic_gluten child153)
     (not_allergic_gluten child345)
     (not_allergic_gluten child165)
     (waiting child345 table31)
     (waiting child153 table91)
     (waiting child165 table131)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child345)
     (served child153)
     (served child165)
    )
  )
)
