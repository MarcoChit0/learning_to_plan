; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 841884

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child279 child79 - child
    bread260 bread52 - bread-portion
    content421 content60 - content-portion
    tray274 tray205 - tray
    table449 table258 table308 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray274 kitchen)
     (at tray205 kitchen)
     (at_kitchen_bread bread260)
     (at_kitchen_bread bread52)
     (at_kitchen_content content421)
     (at_kitchen_content content60)
     (not_allergic_gluten child79)
     (not_allergic_gluten child279)
     (waiting child279 table308)
     (waiting child79 table308)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child279)
     (served child79)
    )
  )
)
