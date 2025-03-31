; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 719152

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child463 child264 - child
    bread58 bread12 - bread-portion
    content378 content89 - content-portion
    tray338 tray214 - tray
    table394 table104 table144 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray338 kitchen)
     (at tray214 kitchen)
     (at_kitchen_bread bread58)
     (at_kitchen_bread bread12)
     (at_kitchen_content content378)
     (at_kitchen_content content89)
     (no_gluten_bread bread58)
     (no_gluten_content content378)
     (allergic_gluten child463)
     (not_allergic_gluten child264)
     (waiting child463 table144)
     (waiting child264 table104)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child463)
     (served child264)
    )
  )
)
