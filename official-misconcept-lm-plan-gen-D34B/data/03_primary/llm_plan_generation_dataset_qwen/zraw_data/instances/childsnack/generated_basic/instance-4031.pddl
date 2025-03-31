; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 654334

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child170 child88 child94 - child
    bread85 bread284 bread264 - bread-portion
    content360 content46 content145 - content-portion
    tray494 - tray
    table175 table464 table127 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray494 kitchen)
     (at_kitchen_bread bread85)
     (at_kitchen_bread bread284)
     (at_kitchen_bread bread264)
     (at_kitchen_content content360)
     (at_kitchen_content content46)
     (at_kitchen_content content145)
     (not_allergic_gluten child170)
     (not_allergic_gluten child88)
     (not_allergic_gluten child94)
     (waiting child170 table464)
     (waiting child88 table127)
     (waiting child94 table175)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child170)
     (served child88)
     (served child94)
    )
  )
)
